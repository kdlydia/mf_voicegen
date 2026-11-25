#include "TorchFormat.hpp"

#include "MayaFlux/IO/Tensor/Detail/TensorCodec.hpp"
#include "MayaFlux/IO/Tensor/TensorFormat.hpp"
#include "MayaFlux/Journal/Archivist.hpp"

namespace MayaFlux::IO {

namespace {

    constexpr uint64_t max_pickle_bytes = 1ULL << 30;
    constexpr uint64_t max_directory_bytes = 1ULL << 28;
    constexpr size_t max_depth = 256;

    /** A malformed archive or pickle; open() turns it into the reason it refuses the file. */
    struct Malformed : std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    uint64_t little(std::span<const std::byte> bytes, size_t offset, size_t width)
    {
        if (offset + width > bytes.size()) {
            throw Malformed("a zip record is truncated");
        }
        uint64_t value = 0;
        for (size_t i = 0; i < width; ++i) {
            value |= static_cast<uint64_t>(std::to_integer<uint8_t>(bytes[offset + i])) << (8 * i);
        }
        return value;
    }

    std::vector<std::byte> read_at(std::ifstream& file, uint64_t offset, uint64_t size)
    {
        std::vector<std::byte> bytes(size);
        file.clear();
        file.seekg(static_cast<std::streamoff>(offset));
        if (!file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size))) {
            throw Malformed("the file ends inside a zip record");
        }
        return bytes;
    }

    // ---------------------------------------------------------------- zip

    struct ZipEntry {
        std::string name;
        uint16_t method {};
        uint64_t size {};
        uint64_t local_offset {};
    };

    std::vector<ZipEntry> read_directory(std::ifstream& file, uint64_t file_size)
    {
        constexpr uint64_t eocd_bytes = 22;
        if (file_size < eocd_bytes) {
            throw Malformed("the file is too short to be a zip archive");
        }
        const uint64_t tail_size = std::min<uint64_t>(file_size, eocd_bytes + 0xFFFF);
        const auto tail = read_at(file, file_size - tail_size, tail_size);

        std::optional<size_t> eocd;
        for (size_t at = tail.size() - eocd_bytes + 1; at-- > 0;) {
            if (little(tail, at, 4) == 0x06054b50) {
                eocd = at;
                break;
            }
        }
        if (!eocd) {
            throw Malformed("no zip end-of-directory record");
        }

        uint64_t entries = little(tail, *eocd + 10, 2);
        uint64_t directory_size = little(tail, *eocd + 12, 4);
        uint64_t directory_offset = little(tail, *eocd + 16, 4);

        if (entries == 0xFFFF || directory_size == 0xFFFFFFFF || directory_offset == 0xFFFFFFFF) {
            const uint64_t eocd_position = file_size - tail_size + *eocd;
            if (eocd_position < 20) {
                throw Malformed("a zip64 archive without its locator");
            }
            const auto locator = read_at(file, eocd_position - 20, 20);
            if (little(locator, 0, 4) != 0x07064b50) {
                throw Malformed("a zip64 archive without its locator");
            }
            const auto record = read_at(file, little(locator, 8, 8), 56);
            if (little(record, 0, 4) != 0x06064b50) {
                throw Malformed("the zip64 end-of-directory record is missing");
            }
            entries = little(record, 32, 8);
            directory_size = little(record, 40, 8);
            directory_offset = little(record, 48, 8);
        }

        if (directory_size > max_directory_bytes || directory_offset + directory_size > file_size) {
            throw Malformed("the zip directory lies outside the file");
        }
        const auto directory = read_at(file, directory_offset, directory_size);

        std::vector<ZipEntry> out;
        out.reserve(static_cast<size_t>(std::min<uint64_t>(entries, 1U << 20)));
        size_t at = 0;
        for (uint64_t i = 0; i < entries; ++i) {
            if (little(directory, at, 4) != 0x02014b50) {
                throw Malformed("a zip directory entry is damaged");
            }
            ZipEntry entry;
            entry.method = static_cast<uint16_t>(little(directory, at + 10, 2));
            uint64_t compressed = little(directory, at + 20, 4);
            entry.size = little(directory, at + 24, 4);
            const size_t name_length = little(directory, at + 28, 2);
            const size_t extra_length = little(directory, at + 30, 2);
            const size_t comment_length = little(directory, at + 32, 2);
            entry.local_offset = little(directory, at + 42, 4);
            if (at + 46 + name_length + extra_length > directory.size()) {
                throw Malformed("a zip directory entry is truncated");
            }
            entry.name.assign(reinterpret_cast<const char*>(directory.data() + at + 46), name_length);

            size_t extra = at + 46 + name_length;
            const size_t extra_end = extra + extra_length;
            while (extra + 4 <= extra_end) {
                const uint64_t id = little(directory, extra, 2);
                const size_t size = little(directory, extra + 2, 2);
                if (id == 0x0001) {
                    size_t field = extra + 4;
                    if (entry.size == 0xFFFFFFFF) {
                        entry.size = little(directory, field, 8);
                        field += 8;
                    }
                    if (compressed == 0xFFFFFFFF) {
                        compressed = little(directory, field, 8);
                        field += 8;
                    }
                    if (entry.local_offset == 0xFFFFFFFF) {
                        entry.local_offset = little(directory, field, 8);
                    }
                }
                extra += 4 + size;
            }
            out.push_back(std::move(entry));
            at += 46 + name_length + extra_length + comment_length;
        }
        return out;
    }

    uint64_t data_offset(std::ifstream& file, const ZipEntry& entry)
    {
        const auto header = read_at(file, entry.local_offset, 30);
        if (little(header, 0, 4) != 0x04034b50) {
            throw Malformed("the local header of '" + entry.name + "' is damaged");
        }
        return entry.local_offset + 30 + little(header, 26, 2) + little(header, 28, 2);
    }

    // ------------------------------------------------------------- pickle

    enum class Kind : uint8_t {
        None,
        Bool,
        Int,
        Float,
        String,
        Bytes,
        List,
        Tuple,
        Dict,
        Global,
        Storage,
        Tensor,
        Opaque
    };

    struct Storage {
        std::string key;
        Kakshya::TensorDType dtype;
        uint64_t numel {};
    };

    struct TensorView {
        Storage storage;
        uint64_t offset {};
        std::vector<uint64_t> shape;
        std::vector<int64_t> strides;
    };

    struct Value;
    using Ref = std::shared_ptr<Value>;

    struct Value {
        Kind kind { Kind::None };
        int64_t integer {};
        double real {};
        std::string text;
        std::vector<Ref> items;
        std::vector<std::pair<Ref, Ref>> entries;
        std::optional<Storage> storage;
        std::optional<TensorView> tensor;
    };

    Ref make(Kind kind)
    {
        auto value = std::make_shared<Value>();
        value->kind = kind;
        return value;
    }

    Ref make_text(Kind kind, std::string text)
    {
        auto value = make(kind);
        value->text = std::move(text);
        return value;
    }

    Ref make_int(int64_t integer)
    {
        auto value = make(Kind::Int);
        value->integer = integer;
        return value;
    }

    std::optional<Kakshya::TensorDType> storage_dtype(std::string_view global)
    {
        using Kakshya::TensorDType;
        static const std::array<std::pair<std::string_view, TensorDType>, 13> table { {
            { "torch DoubleStorage", TensorDType::of<double>() },
            { "torch FloatStorage", TensorDType::of<float>() },
            { "torch HalfStorage", TensorDType::float16() },
            { "torch BFloat16Storage", TensorDType::bfloat16() },
            { "torch LongStorage", TensorDType::of<int64_t>() },
            { "torch IntStorage", TensorDType::of<int32_t>() },
            { "torch ShortStorage", TensorDType::of<int16_t>() },
            { "torch CharStorage", TensorDType::of<int8_t>() },
            { "torch ByteStorage", TensorDType::of<uint8_t>() },
            { "torch BoolStorage", TensorDType::of<bool>() },
            { "torch ComplexFloatStorage", TensorDType::of<std::complex<float>>() },
            { "torch ComplexDoubleStorage", TensorDType::of<std::complex<double>>() },
            { "torch UntypedStorage", TensorDType::of<uint8_t>() },
        } };
        for (const auto& [name, dtype] : table) {
            if (name == global) {
                return dtype;
            }
        }
        return std::nullopt;
    }

    int64_t as_int(const Ref& value, std::string_view what)
    {
        if (!value || (value->kind != Kind::Int && value->kind != Kind::Bool)) {
            throw Malformed(std::string(what) + " is not an integer");
        }
        return value->integer;
    }

    const std::vector<Ref>& as_sequence(const Ref& value, std::string_view what)
    {
        if (!value || (value->kind != Kind::Tuple && value->kind != Kind::List)) {
            throw Malformed(std::string(what) + " is not a tuple");
        }
        return value->items;
    }

    /**
     * A pickle machine that only builds values. Globals stay names; calling one
     * is understood for the few torch.save uses to describe tensors and
     * containers, and gives an opaque value for anything else.
     */
    class Unpickler {
    public:
        explicit Unpickler(std::span<const std::byte> data)
            : m_data(data)
        {
        }

        Ref run()
        {
            while (true) {
                const auto op = static_cast<uint8_t>(byte());
                if (op == '.') {
                    return pop();
                }
                step(op);
            }
        }

    private:
        std::span<const std::byte> m_data;
        size_t m_pos {};
        std::vector<Ref> m_stack;
        std::vector<size_t> m_marks;
        std::unordered_map<uint64_t, Ref> m_memo;

        std::byte byte()
        {
            if (m_pos >= m_data.size()) {
                throw Malformed("data.pkl ends before its STOP");
            }
            return m_data[m_pos++];
        }

        uint64_t unsigned_le(size_t width)
        {
            if (m_data.size() - m_pos < width) {
                throw Malformed("data.pkl ends inside an argument");
            }
            const uint64_t value = little(m_data, m_pos, width);
            m_pos += width;
            return value;
        }

        std::string raw(uint64_t length)
        {
            if (m_data.size() - m_pos < length) {
                throw Malformed("data.pkl ends inside a string");
            }
            std::string text(reinterpret_cast<const char*>(m_data.data() + m_pos), static_cast<size_t>(length));
            m_pos += static_cast<size_t>(length);
            return text;
        }

        std::string line()
        {
            std::string text;
            while (true) {
                const auto c = static_cast<char>(byte());
                if (c == '\n') {
                    return text;
                }
                text.push_back(c);
            }
        }

        static int64_t parse_integer(std::string text)
        {
            while (!text.empty() && (text.back() == 'L' || text.back() == '\r')) {
                text.pop_back();
            }
            int64_t value {};
            const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
            if (error != std::errc {} || end != text.data() + text.size()) {
                throw Malformed("data.pkl holds an integer that is not a 64-bit number: " + text);
            }
            return value;
        }

        void push(Ref value) { m_stack.push_back(std::move(value)); }

        Ref pop()
        {
            if (m_stack.empty() || (!m_marks.empty() && m_stack.size() == m_marks.back())) {
                throw Malformed("data.pkl pops an empty stack");
            }
            Ref value = std::move(m_stack.back());
            m_stack.pop_back();
            return value;
        }

        Ref& top()
        {
            if (m_stack.empty()) {
                throw Malformed("data.pkl reads an empty stack");
            }
            return m_stack.back();
        }

        std::vector<Ref> pop_mark()
        {
            if (m_marks.empty() || m_marks.back() > m_stack.size()) {
                throw Malformed("data.pkl closes a mark it never opened");
            }
            const size_t mark = m_marks.back();
            m_marks.pop_back();
            std::vector<Ref> items(std::make_move_iterator(m_stack.begin() + static_cast<std::ptrdiff_t>(mark)),
                std::make_move_iterator(m_stack.end()));
            m_stack.resize(mark);
            return items;
        }

        Ref fetch(uint64_t index)
        {
            const auto found = m_memo.find(index);
            if (found == m_memo.end()) {
                throw Malformed("data.pkl reads memo slot " + std::to_string(index) + " before writing it");
            }
            return found->second;
        }

        static Ref tuple(std::vector<Ref> items)
        {
            auto value = make(Kind::Tuple);
            value->items = std::move(items);
            return value;
        }

        static void set_items(const Ref& target, std::vector<Ref> pairs)
        {
            if (pairs.size() % 2 != 0) {
                throw Malformed("data.pkl sets a key without a value");
            }
            for (size_t i = 0; i < pairs.size(); i += 2) {
                if (target->kind == Kind::Dict) {
                    target->entries.emplace_back(std::move(pairs[i]), std::move(pairs[i + 1]));
                }
            }
        }

        static void append_items(const Ref& target, std::vector<Ref> items)
        {
            if (target->kind == Kind::List) {
                for (auto& item : items) {
                    target->items.push_back(std::move(item));
                }
            }
        }

        static Ref persistent(const Ref& id)
        {
            const auto& fields = as_sequence(id, "a persistent id");
            if (fields.size() < 5 || fields[0]->kind != Kind::String || fields[0]->text != "storage") {
                throw Malformed("data.pkl names a persistent object that is not a tensor storage");
            }
            if (fields[1]->kind != Kind::Global || fields[2]->kind != Kind::String) {
                throw Malformed("a storage record lacks its type or key");
            }
            const auto dtype = storage_dtype(fields[1]->text);
            if (!dtype) {
                throw Malformed("storage type '" + fields[1]->text + "' is not supported");
            }
            auto value = make(Kind::Storage);
            value->storage = Storage { .key = fields[2]->text, .dtype = *dtype,
                .numel = static_cast<uint64_t>(as_int(fields[4], "a storage size")) };
            return value;
        }

        static Ref rebuild_tensor(const std::vector<Ref>& args)
        {
            if (args.size() < 4 || !args[0] || args[0]->kind != Kind::Storage) {
                throw Malformed("a tensor record lacks its storage, offset, size or stride");
            }
            TensorView view { .storage = *args[0]->storage };
            const int64_t offset = as_int(args[1], "a storage offset");
            if (offset < 0) {
                throw Malformed("a tensor has a negative storage offset");
            }
            view.offset = static_cast<uint64_t>(offset);
            for (const auto& extent : as_sequence(args[2], "a tensor size")) {
                const int64_t size = as_int(extent, "a tensor extent");
                if (size < 0) {
                    throw Malformed("a tensor has a negative extent");
                }
                view.shape.push_back(static_cast<uint64_t>(size));
            }
            for (const auto& stride : as_sequence(args[3], "a tensor stride")) {
                view.strides.push_back(as_int(stride, "a tensor stride"));
            }
            if (view.strides.size() != view.shape.size()) {
                throw Malformed("a tensor's size and stride differ in rank");
            }
            auto value = make(Kind::Tensor);
            value->tensor = std::move(view);
            return value;
        }

        static Ref call(const Ref& callable, const Ref& arguments)
        {
            if (!callable || callable->kind != Kind::Global) {
                return make(Kind::Opaque);
            }
            const std::string& name = callable->text;
            const auto& args = arguments && (arguments->kind == Kind::Tuple || arguments->kind == Kind::List)
                ? arguments->items
                : std::vector<Ref> {};

            if (name == "collections OrderedDict" || name == "builtins dict" || name == "__builtin__ dict") {
                auto dict = make(Kind::Dict);
                if (!args.empty() && args[0] && (args[0]->kind == Kind::List || args[0]->kind == Kind::Tuple)) {
                    for (const auto& pair : args[0]->items) {
                        if (pair && pair->kind == Kind::Tuple && pair->items.size() == 2) {
                            dict->entries.emplace_back(pair->items[0], pair->items[1]);
                        }
                    }
                }
                return dict;
            }
            if (name == "torch._utils _rebuild_tensor_v2" || name == "torch._utils _rebuild_tensor") {
                return rebuild_tensor(args);
            }
            if (name == "torch._utils _rebuild_parameter" || name == "torch._utils _rebuild_parameter_with_state") {
                return args.empty() ? make(Kind::Opaque) : args[0];
            }
            if (name == "torch._tensor _rebuild_from_type_v2" && args.size() >= 3) {
                return call(args[0], args[2]);
            }
            if (name == "torch Size" || name == "builtins tuple" || name == "builtins list"
                || name == "builtins set" || name == "builtins frozenset") {
                auto sequence = make(name == "builtins list" ? Kind::List : Kind::Tuple);
                if (!args.empty() && args[0]) {
                    sequence->items = args[0]->items;
                }
                return sequence;
            }
            return make(Kind::Opaque);
        }

        void step(uint8_t op)
        {
            switch (op) {
            case 0x80: // PROTO
                static_cast<void>(byte());
                break;
            case 0x95: // FRAME
                static_cast<void>(unsigned_le(8));
                break;
            case '(': // MARK
                m_marks.push_back(m_stack.size());
                break;
            case '0': // POP
                static_cast<void>(pop());
                break;
            case '1': // POP_MARK
                static_cast<void>(pop_mark());
                break;
            case '2': // DUP
                push(top());
                break;
            case 'N':
                push(make(Kind::None));
                break;
            case 0x88:
            case 0x89: {
                auto value = make(Kind::Bool);
                value->integer = op == 0x88 ? 1 : 0;
                push(value);
                break;
            }
            case 'J':
                push(make_int(static_cast<int32_t>(unsigned_le(4))));
                break;
            case 'K':
                push(make_int(static_cast<int64_t>(unsigned_le(1))));
                break;
            case 'M':
                push(make_int(static_cast<int64_t>(unsigned_le(2))));
                break;
            case 0x8a: // LONG1
            case 0x8b: { // LONG4
                const uint64_t length = unsigned_le(op == 0x8a ? 1 : 4);
                if (length > 8) {
                    static_cast<void>(raw(length));
                    push(make(Kind::Opaque));
                    break;
                }
                uint64_t bits = length == 0 ? 0 : unsigned_le(static_cast<size_t>(length));
                if (length > 0 && length < 8 && (bits >> (8 * length - 1)) & 1U) {
                    bits |= ~0ULL << (8 * length);
                }
                push(make_int(static_cast<int64_t>(bits)));
                break;
            }
            case 'I': {
                const std::string text = line();
                if (text == "01" || text == "00") {
                    auto value = make(Kind::Bool);
                    value->integer = text == "01" ? 1 : 0;
                    push(value);
                } else {
                    push(make_int(parse_integer(text)));
                }
                break;
            }
            case 'L':
                push(make_int(parse_integer(line())));
                break;
            case 'G': {
                uint64_t bits = 0;
                for (int i = 0; i < 8; ++i) {
                    bits = (bits << 8) | std::to_integer<uint8_t>(byte());
                }
                auto value = make(Kind::Float);
                value->real = std::bit_cast<double>(bits);
                push(value);
                break;
            }
            case 'F': {
                auto value = make(Kind::Float);
                const std::string text = line();
                std::from_chars(text.data(), text.data() + text.size(), value->real);
                push(value);
                break;
            }
            case 'X':
                push(make_text(Kind::String, raw(unsigned_le(4))));
                break;
            case 0x8c:
                push(make_text(Kind::String, raw(unsigned_le(1))));
                break;
            case 0x8d:
                push(make_text(Kind::String, raw(unsigned_le(8))));
                break;
            case 'V':
                push(make_text(Kind::String, line()));
                break;
            case 'S': {
                std::string text = line();
                if (text.size() >= 2 && (text.front() == '\'' || text.front() == '"')) {
                    text = text.substr(1, text.size() - 2);
                }
                push(make_text(Kind::String, std::move(text)));
                break;
            }
            case 'T':
                push(make_text(Kind::String, raw(unsigned_le(4))));
                break;
            case 'U':
                push(make_text(Kind::String, raw(unsigned_le(1))));
                break;
            case 'B':
                push(make_text(Kind::Bytes, raw(unsigned_le(4))));
                break;
            case 'C':
                push(make_text(Kind::Bytes, raw(unsigned_le(1))));
                break;
            case 0x8e:
            case 0x96:
                push(make_text(Kind::Bytes, raw(unsigned_le(8))));
                break;
            case ')':
                push(tuple({}));
                break;
            case 't':
                push(tuple(pop_mark()));
                break;
            case 0x85:
            case 0x86:
            case 0x87: {
                std::vector<Ref> items(static_cast<size_t>(op - 0x84));
                for (auto it = items.rbegin(); it != items.rend(); ++it) {
                    *it = pop();
                }
                push(tuple(std::move(items)));
                break;
            }
            case ']':
            case 0x8f: // EMPTY_SET
                push(make(Kind::List));
                break;
            case 'l':
            case 0x91: { // FROZENSET
                auto list = make(Kind::List);
                list->items = pop_mark();
                push(list);
                break;
            }
            case 'a': {
                Ref item = pop();
                append_items(top(), { std::move(item) });
                break;
            }
            case 'e':
            case 0x90: { // ADDITEMS
                auto items = pop_mark();
                append_items(top(), std::move(items));
                break;
            }
            case '}':
                push(make(Kind::Dict));
                break;
            case 'd': {
                auto dict = make(Kind::Dict);
                set_items(dict, pop_mark());
                push(dict);
                break;
            }
            case 's': {
                Ref value = pop();
                Ref key = pop();
                set_items(top(), { std::move(key), std::move(value) });
                break;
            }
            case 'u': {
                auto pairs = pop_mark();
                set_items(top(), std::move(pairs));
                break;
            }
            case 'c': {
                std::string module = line();
                push(make_text(Kind::Global, module + " " + line()));
                break;
            }
            case 0x93: { // STACK_GLOBAL
                const Ref name = pop();
                const Ref module = pop();
                if (name->kind != Kind::String || module->kind != Kind::String) {
                    throw Malformed("STACK_GLOBAL is not given two strings");
                }
                push(make_text(Kind::Global, module->text + " " + name->text));
                break;
            }
            case 'R': {
                const Ref arguments = pop();
                const Ref callable = pop();
                push(call(callable, arguments));
                break;
            }
            case 0x81: { // NEWOBJ
                const Ref arguments = pop();
                const Ref cls = pop();
                push(call(cls, arguments));
                break;
            }
            case 0x92: { // NEWOBJ_EX
                static_cast<void>(pop());
                const Ref arguments = pop();
                const Ref cls = pop();
                push(call(cls, arguments));
                break;
            }
            case 'b': { // BUILD
                Ref state = pop();
                if (top()->kind == Kind::Opaque) {
                    top()->items.push_back(std::move(state));
                }
                break;
            }
            case 'i': { // INST
                static_cast<void>(line());
                static_cast<void>(line());
                static_cast<void>(pop_mark());
                push(make(Kind::Opaque));
                break;
            }
            case 'o': // OBJ
                static_cast<void>(pop_mark());
                push(make(Kind::Opaque));
                break;
            case 'Q': // BINPERSID
                push(persistent(pop()));
                break;
            case 'P':
                throw Malformed("data.pkl uses a text persistent id, which torch.save does not write");
            case 'p':
                m_memo[static_cast<uint64_t>(parse_integer(line()))] = top();
                break;
            case 'q':
                m_memo[unsigned_le(1)] = top();
                break;
            case 'r':
                m_memo[unsigned_le(4)] = top();
                break;
            case 0x94: // MEMOIZE
                m_memo[m_memo.size()] = top();
                break;
            case 'g':
                push(fetch(static_cast<uint64_t>(parse_integer(line()))));
                break;
            case 'h':
                push(fetch(unsigned_le(1)));
                break;
            case 'j':
                push(fetch(unsigned_le(4)));
                break;
            default:
                throw Malformed(std::format("data.pkl uses pickle opcode 0x{:02x}, which this reader does not interpret", op));
            }
        }
    };

    // -------------------------------------------------------------- walk

    struct Found {
        std::vector<std::pair<std::string, TensorView>> tensors;
        std::map<std::string, std::string> metadata;
    };

    std::string joined(const std::string& path, const std::string& part)
    {
        return path.empty() ? part : path + "." + part;
    }

    void collect(const Ref& value, const std::string& path, Found& found, std::vector<const Value*>& active)
    {
        if (!value || active.size() > max_depth || std::ranges::find(active, value.get()) != active.end()) {
            return;
        }
        active.push_back(value.get());
        switch (value->kind) {
        case Kind::Tensor:
            found.tensors.emplace_back(path, *value->tensor);
            break;
        case Kind::Dict:
            for (const auto& [key, item] : value->entries) {
                if (key && key->kind == Kind::String) {
                    collect(item, joined(path, key->text), found, active);
                } else if (key && key->kind == Kind::Int) {
                    collect(item, joined(path, std::to_string(key->integer)), found, active);
                }
            }
            break;
        case Kind::List:
        case Kind::Tuple:
            for (size_t i = 0; i < value->items.size(); ++i) {
                collect(value->items[i], joined(path, std::to_string(i)), found, active);
            }
            break;
        case Kind::Opaque:
            for (const auto& state : value->items) {
                collect(state, path, found, active);
            }
            break;
        case Kind::Bool:
            if (!path.empty()) {
                found.metadata[path] = value->integer != 0 ? "true" : "false";
            }
            break;
        case Kind::Int:
            if (!path.empty()) {
                found.metadata[path] = std::to_string(value->integer);
            }
            break;
        case Kind::Float:
            if (!path.empty()) {
                found.metadata[path] = std::format("{}", value->real);
            }
            break;
        case Kind::String:
            if (!path.empty()) {
                found.metadata[path] = value->text;
            }
            break;
        default:
            break;
        }
        active.pop_back();
    }

    std::string_view prefix_of(std::string_view pickle_name)
    {
        return pickle_name.substr(0, pickle_name.size() - std::string_view("data.pkl").size());
    }

}

void TorchCheckpointReader::register_with_registry()
{
    TensorFormatRegistry::instance().register_format(TensorFormat {
        .name = "torch",
        .extensions = { "pt", "pth", "ckpt" },
        .probe = &TorchCheckpointReader::probe,
        .make_reader = []() -> std::unique_ptr<TensorReader> { return std::make_unique<TorchCheckpointReader>(); },
        .make_writer = nullptr });

    MF_INFO(Journal::Component::IO, Journal::Context::Init,
        "TorchCheckpointReader registered for: pt, pth, ckpt and torch .bin archives");
}

bool TorchCheckpointReader::probe(std::span<const std::byte> head, uint64_t file_size)
{
    if (head.size() < 30 || file_size < 30 || little(head, 0, 4) != 0x04034b50) {
        return false;
    }
    const size_t name_length = little(head, 26, 2);
    if (30 + name_length > head.size()) {
        return false;
    }
    const std::string_view name(reinterpret_cast<const char*>(head.data() + 30), name_length);
    return name == "data.pkl" || name.ends_with("/data.pkl");
}

bool TorchCheckpointReader::can_read(const std::string& filepath) const
{
    const auto head = Detail::read_file_head(resolve_path(filepath), 4096);
    return head && probe(head->bytes, head->file_size);
}

bool TorchCheckpointReader::reject(std::string reason)
{
    close();
    return fail(std::move(reason));
}

void TorchCheckpointReader::close()
{
    if (m_file.is_open()) {
        m_file.close();
    }
    m_entries.clear();
    m_metadata.clear();
    m_big_endian = false;
    m_open = false;
}

bool TorchCheckpointReader::open(const std::string& filepath, FileReadOptions)
{
    close();
    m_path = resolve_path(filepath);
    m_file.open(m_path, std::ios::binary);
    if (!m_file) {
        return reject("cannot open '" + m_path + "'");
    }
    m_file.seekg(0, std::ios::end);
    const auto file_size = static_cast<uint64_t>(m_file.tellg());

    try {
        const auto lead = read_at(m_file, 0, std::min<uint64_t>(file_size, 2));
        if (lead.size() == 2 && std::to_integer<uint8_t>(lead[0]) == 0x80) {
            return reject("'" + m_path + "' is a bare pickle, the format torch.save wrote before PyTorch 1.6; load and save it again with a current PyTorch");
        }

        const auto directory = read_directory(m_file, file_size);
        const auto pickle = std::ranges::find_if(directory, [](const ZipEntry& entry) {
            return entry.name == "data.pkl" || entry.name.ends_with("/data.pkl");
        });
        if (pickle == directory.end()) {
            return reject("'" + m_path + "' is a zip archive without a data.pkl, so not a PyTorch checkpoint");
        }
        const std::string prefix(prefix_of(pickle->name));

        const auto entry_named = [&](const std::string& name) -> const ZipEntry* {
            const auto found = std::ranges::find_if(directory, [&](const ZipEntry& entry) { return entry.name == name; });
            return found == directory.end() ? nullptr : &*found;
        };
        const auto stored = [&](const ZipEntry& entry) {
            if (entry.method != 0) {
                throw Malformed("entry '" + entry.name + "' is compressed; torch.save stores entries uncompressed");
            }
        };

        stored(*pickle);
        if (pickle->size > max_pickle_bytes) {
            return reject("data.pkl is larger than this reader accepts");
        }
        const auto pickle_bytes = read_at(m_file, data_offset(m_file, *pickle), pickle->size);

        if (const auto* order = entry_named(prefix + "byteorder")) {
            stored(*order);
            const auto bytes = read_at(m_file, data_offset(m_file, *order), std::min<uint64_t>(order->size, 16));
            m_big_endian = std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()).starts_with("big");
        }

        Unpickler machine(pickle_bytes);
        const Ref root = machine.run();

        Found found;
        std::vector<const Value*> active;
        collect(root, {}, found, active);
        m_metadata = std::move(found.metadata);

        std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> storages;
        for (auto& [name, view] : found.tensors) {
            if (name.empty() || std::ranges::any_of(m_entries, [&](const Entry& e) { return e.spec.name == name; })) {
                MF_WARN(Journal::Component::IO, Journal::Context::FileIO,
                    "TorchCheckpointReader: skipping a tensor without a unique name in '{}'", m_path);
                continue;
            }
            auto known = storages.find(view.storage.key);
            if (known == storages.end()) {
                const auto* data = entry_named(prefix + "data/" + view.storage.key);
                if (data == nullptr) {
                    return reject("storage '" + view.storage.key + "' of tensor '" + name + "' is missing from the archive");
                }
                stored(*data);
                known = storages.emplace(view.storage.key, std::pair { data_offset(m_file, *data), data->size }).first;
            }

            const size_t element_bytes = view.storage.dtype.element_bytes();
            Entry entry;
            entry.spec = Kakshya::TensorSpec::contiguous(view.storage.dtype, view.shape, name);
            entry.strides = view.strides;
            entry.storage_offset = view.offset;
            entry.data_offset = known->second.first;
            entry.data_size = known->second.second;

            Kakshya::TensorSpec addressed = entry.spec;
            addressed.strides = view.strides;
            addressed.byte_offset = view.offset * element_bytes;
            const auto range = addressed.byte_range();
            if (const auto why = addressed.diagnose(); !why.empty()) {
                return reject("tensor '" + name + "': " + why);
            }
            if (!range || range->second > entry.data_size) {
                return reject("tensor '" + name + "' reaches past the end of its storage");
            }
            m_entries.push_back(std::move(entry));
        }
    } catch (const Malformed& problem) {
        return reject("'" + m_path + "': " + problem.what());
    }

    m_open = true;
    m_last_error.clear();
    return true;
}

std::vector<Kakshya::TensorSpec> TorchCheckpointReader::describe() const
{
    std::vector<Kakshya::TensorSpec> specs;
    specs.reserve(m_entries.size());
    for (const auto& entry : m_entries) {
        specs.push_back(entry.spec);
    }
    return specs;
}

std::optional<Kakshya::TensorData> TorchCheckpointReader::read(std::string_view name)
{
    if (!m_open) {
        fail("no file is open");
        return std::nullopt;
    }
    const auto entry = std::ranges::find_if(m_entries, [name](const Entry& e) { return e.spec.name == name; });
    if (entry == m_entries.end()) {
        fail("no tensor named '" + std::string(name) + "'");
        return std::nullopt;
    }

    Kakshya::TensorSpec view = entry->spec;
    view.strides = entry->strides;
    view.byte_offset = entry->storage_offset * view.dtype.element_bytes();
    const auto range = view.byte_range();
    if (!range) {
        fail("tensor '" + std::string(name) + "' has no addressable range");
        return std::nullopt;
    }
    if (range->second == range->first) {
        return Kakshya::TensorData::allocate(entry->spec);
    }

    std::vector<std::byte> bytes;
    try {
        bytes = read_at(m_file, entry->data_offset + range->first, range->second - range->first);
    } catch (const Malformed& problem) {
        fail("tensor '" + std::string(name) + "': " + problem.what());
        return std::nullopt;
    }
    view.byte_offset -= range->first;
    if (m_big_endian == Detail::host_is_little_endian()) {
        Detail::swap_components(bytes, Detail::component_bytes(view.dtype));
    }

    const bool dense = view.byte_offset == 0 && view.is_contiguous();
    auto tensor = Kakshya::TensorData::from_bytes(dense ? entry->spec : view, bytes);
    if (!tensor) {
        fail("tensor '" + std::string(name) + "' could not be built from its storage");
        return std::nullopt;
    }
    if (dense) {
        return tensor;
    }
    const auto access = tensor->access();
    if (!access) {
        fail("tensor '" + std::string(name) + "' could not be gathered");
        return std::nullopt;
    }
    return Kakshya::TensorData::copy_of(*access);
}

}
